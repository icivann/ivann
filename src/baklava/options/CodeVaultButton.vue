<template>
  <div class="d-sm-flex">
    <UIButton text="Select Function" :primary="true" class="m-auto"
              @click="enterCodeVault(option.customNode)"/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Mutation } from 'vuex-class';
import Custom from '@/nodes/model/custom/Custom';
import UIButton from '@/components/buttons/UIButton.vue';

@Component({
  components: { UIButton },
})
export default class CodeVaultButton extends Vue {
  @Mutation('enterCodeVault') enterCodeVault!: (node?: Custom) => void;
  @Prop() value!: number;

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: {
    customNode?: Custom;
  };
}
</script>
