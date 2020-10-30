<template>
  <span @click="saveModel(index)" class="save-button">save</span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { mapMutations } from 'vuex';

@Component({
  methods: mapMutations(['saveModel']),
})
export default class SaveEditorButton extends Vue {
  @Prop({ required: true }) readonly index!: number;
}
</script>

<style lang="scss" scoped>
.save-button {
  color: var(--foreground);
  background: none;
  padding: 0 4px 2px;
  border-radius: 2px;

  &:hover {
    background: var(--blue);
  }
}
</style>
