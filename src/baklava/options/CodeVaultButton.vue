<template>
  <div class="d-sm-flex">
    <UIButton text="Select Function" :primary="true" class="m-auto"
              @click="toCodeVault(option.customNode)"/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Mutation } from 'vuex-class';
import Custom from '@/nodes/common/Custom';
import UIButton from '@/components/buttons/UIButton.vue';

@Component({
  components: { UIButton },
})
export default class CodeVaultButton extends Vue {
  @Mutation('enterCodeVault') enterCodeVault!: () => void;
  @Mutation('linkNode') linkNode!: (node?: Custom) => void;
  @Prop() value!: number;

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: {
    customNode?: Custom;
  };

  private toCodeVault(node: Custom) {
    this.linkNode(node);
    this.enterCodeVault();
  }
}
</script>
